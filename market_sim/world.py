"""Authored content: all 61 hand-crafted items and all production rules.

Importing this module registers everything into globalItemIndex and
productionGraphs automatically (module-level side effects, run once).

Depends on: types, structures, _globals
"""
from .types      import item, itemTag, itemFlag, qualityTier
from .structures import productionRule, registerProductionRule
from ._globals   import globalItemIndex


# =============================================================================
# ITEM DEFINITIONS  (Early Medieval → Late Industrial)
# =============================================================================

# ── Early Medieval (level 1) ──────────────────────────────────────────────────
wheat        = item(1,  {itemTag.crops, itemTag.food},
                        {itemFlag.isNutritional, itemFlag.isPerishable},                                      1, True,  "wheat")
rye          = item(2,  {itemTag.crops, itemTag.food},
                        {itemFlag.isNutritional, itemFlag.isPerishable},                                      1, True,  "rye")
bread        = item(3,  {itemTag.food},
                        {itemFlag.isNutritional, itemFlag.isPerishable},                                      1, False, "bread")
wool_raw     = item(4,  {itemTag.livestock, itemTag.fiber},
                        {itemFlag.isFlexible, itemFlag.isRough},                                              1, True,  "raw_wool")
woolen_fabric= item(5,  {itemTag.fabric},
                        {itemFlag.isDurable, itemFlag.isFlexible},                                            1, False, "woolen_fabric")
basic_clothes= item(6,  {itemTag.consumable},
                        {itemFlag.isDurable, itemFlag.isValuable},                                            1, False, "basic_clothes")
timber       = item(7,  {itemTag.lumber, itemTag.miscellaneous},
                        {itemFlag.isDurable, itemFlag.isTough, itemFlag.isBulky},                             1, True,  "timber")
charcoal     = item(8,  {itemTag.fuel},
                        {itemFlag.isFlammable},                                                               1, False, "charcoal")
iron_ore     = item(9,  {itemTag.ore},
                        {itemFlag.isDurable, itemFlag.isBulky},                                               1, True,  "iron_ore")
iron_ingot   = item(10, {itemTag.metal},
                        {itemFlag.isDurable, itemFlag.isValuable, itemFlag.isRefined},                        1, False, "iron_ingot")
iron_tool    = item(11, {itemTag.tools},
                        {itemFlag.isDurable, itemFlag.isValuable},                                            1, False, "iron_tool")
iron_sword   = item(12, {itemTag.weapons},
                        {itemFlag.isDurable, itemFlag.isValuable},                                            1, False, "iron_sword")
hide         = item(13, {itemTag.livestock, itemTag.leather},
                        {itemFlag.isRough, itemFlag.isPerishable},                                            1, True,  "hide")
leather_item = item(14, {itemTag.leather, itemTag.fabric},
                        {itemFlag.isDurable, itemFlag.isFlexible, itemFlag.isRefined},                        1, False, "leather")
salt         = item(15, {itemTag.spice},
                        {itemFlag.isDurable, itemFlag.isValuable, itemFlag.isRare},                           1, True,  "salt")
clay_pots    = item(16, {itemTag.ceramic, itemTag.consumable},
                        {itemFlag.isFragile},                                                                 1, False, "clay_pots")
ale          = item(17, {itemTag.food, itemTag.alcohol},
                        {itemFlag.isNutritional, itemFlag.isPerishable},                                      1, False, "ale")
mead         = item(18, {itemTag.food, itemTag.alcohol},
                        {itemFlag.isNutritional, itemFlag.isPerishable, itemFlag.isValuable},                 1, False, "mead")
wax_candle   = item(19, {itemTag.consumable, itemTag.fuel},
                        {itemFlag.isPerishable, itemFlag.isFlammable},                                        1, False, "wax_candle")

# ── Medieval (level 2) ────────────────────────────────────────────────────────
cotton_raw   = item(20, {itemTag.crops, itemTag.fiber},
                        {itemFlag.isSmooth, itemFlag.isFlexible, itemFlag.isPerishable},                      2, True,  "raw_cotton")
cotton_fabric= item(21, {itemTag.fabric},
                        {itemFlag.isDurable, itemFlag.isSmooth, itemFlag.isFlexible},                         2, False, "cotton_fabric")
fine_clothes = item(22, {itemTag.consumable, itemTag.luxury},
                        {itemFlag.isDurable, itemFlag.isSmooth, itemFlag.isValuable, itemFlag.isLuxury},      2, False, "fine_clothes")
flax         = item(23, {itemTag.crops, itemTag.fiber},
                        {itemFlag.isFlexible, itemFlag.isPerishable},                                         2, True,  "flax")
linen_fabric = item(24, {itemTag.fabric},
                        {itemFlag.isDurable, itemFlag.isSmooth, itemFlag.isFlexible},                         2, False, "linen_fabric")
copper_ore   = item(25, {itemTag.ore},
                        {itemFlag.isDurable, itemFlag.isBulky},                                               2, True,  "copper_ore")
copper_ingot = item(26, {itemTag.metal},
                        {itemFlag.isDurable, itemFlag.isValuable, itemFlag.isRefined, itemFlag.isConductive}, 2, False, "copper_ingot")
tin_ore      = item(27, {itemTag.ore},
                        {itemFlag.isDurable},                                                                 2, True,  "tin_ore")
bronze_ingot = item(28, {itemTag.metal, itemTag.alloy},
                        {itemFlag.isDurable, itemFlag.isValuable, itemFlag.isRefined,
                         itemFlag.isAlloyed, itemFlag.isHardened},                                            2, False, "bronze_ingot")
crossbow     = item(29, {itemTag.weapons},
                        {itemFlag.isDurable, itemFlag.isValuable},                                            2, False, "crossbow")
arrows       = item(30, {itemTag.ammunition},
                        {itemFlag.isPerishable},                                                              2, False, "arrows")
indigo_dye   = item(31, {itemTag.dye},
                        {itemFlag.isValuable, itemFlag.isLiquid, itemFlag.isRare},                            2, True,  "indigo_dye")
parchment    = item(32, {itemTag.paper},
                        {itemFlag.isDurable, itemFlag.isFragile},                                             2, False, "parchment")
quarry_stone = item(33, {itemTag.stone},
                        {itemFlag.isDurable, itemFlag.isTough, itemFlag.isBulky},                             2, True,  "quarry_stone")
brick        = item(34, {itemTag.ceramic, itemTag.stone},
                        {itemFlag.isDurable, itemFlag.isTough},                                               2, False, "brick")
tar          = item(35, {itemTag.adhesive, itemTag.fuel},
                        {itemFlag.isFlammable, itemFlag.isLiquid},                                            2, True,  "tar")

# ── Early Modern (level 3) ────────────────────────────────────────────────────
silk_thread  = item(36, {itemTag.fiber, itemTag.luxury},
                        {itemFlag.isSmooth, itemFlag.isFlexible, itemFlag.isValuable,
                         itemFlag.isLuxury, itemFlag.isRare},                                                 3, True,  "silk_thread")
silk_fabric  = item(37, {itemTag.fabric, itemTag.luxury},
                        {itemFlag.isDurable, itemFlag.isSmooth, itemFlag.isFlexible,
                         itemFlag.isValuable, itemFlag.isLuxury, itemFlag.isRare},                            3, False, "silk_fabric")
silk_clothes = item(38, {itemTag.consumable, itemTag.luxury},
                        {itemFlag.isDurable, itemFlag.isSmooth, itemFlag.isValuable,
                         itemFlag.isLuxury, itemFlag.isRare},                                                 3, False, "silk_clothes")
steel_ingot  = item(39, {itemTag.metal, itemTag.alloy},
                        {itemFlag.isDurable, itemFlag.isValuable, itemFlag.isRefined,
                         itemFlag.isAlloyed, itemFlag.isHardened, itemFlag.isPure},                           3, False, "steel_ingot")
steel_tool   = item(40, {itemTag.tools},
                        {itemFlag.isDurable, itemFlag.isHardened, itemFlag.isValuable},                       3, False, "steel_tool")
steel_sword  = item(41, {itemTag.weapons},
                        {itemFlag.isDurable, itemFlag.isHardened, itemFlag.isValuable},                       3, False, "steel_sword")
saltpeter_item  = item(42, {itemTag.saltpeter, itemTag.chemical},
                        {itemFlag.isReactive, itemFlag.isPowdered, itemFlag.isValuable},                      3, True,  "saltpeter")
sulfur_item     = item(43, {itemTag.sulfur, itemTag.chemical},
                        {itemFlag.isReactive, itemFlag.isPowdered, itemFlag.isFlammable},                     3, True,  "sulfur")
gunpowder       = item(44, {itemTag.explosive, itemTag.chemical},
                        {itemFlag.isReactive, itemFlag.isPowdered,
                         itemFlag.isFlammable, itemFlag.isValuable},                                          3, False, "gunpowder")
musket_ball  = item(45, {itemTag.ammunition},
                        {itemFlag.isDurable, itemFlag.isValuable},                                            3, False, "musket_ball")
musket       = item(46, {itemTag.weapons},
                        {itemFlag.isDurable, itemFlag.isValuable},                                            3, False, "musket")
glass_item   = item(47, {itemTag.glass},
                        {itemFlag.isFragile, itemFlag.isValuable},                                            3, False, "glass")
rag_paper    = item(48, {itemTag.paper},
                        {itemFlag.isPerishable, itemFlag.isFragile},                                          3, False, "rag_paper")

# ── Early Industrial (level 4) ────────────────────────────────────────────────
coal_item    = item(49, {itemTag.ore, itemTag.fuel, itemTag.coal},
                        {itemFlag.isFlammable, itemFlag.isHighCalorie, itemFlag.isBulky},                     4, True,  "coal")
coke_item    = item(50, {itemTag.coke, itemTag.fuel},
                        {itemFlag.isFlammable, itemFlag.isHighCalorie, itemFlag.isRefined},                   4, False, "coke")
cast_iron    = item(51, {itemTag.metal},
                        {itemFlag.isDurable, itemFlag.isHardened,
                         itemFlag.isBrittle, itemFlag.isRefined},                                             4, False, "cast_iron")
machine_part = item(52, {itemTag.machinery, itemTag.metal},
                        {itemFlag.isDurable, itemFlag.isHardened, itemFlag.isValuable},                       4, False, "machine_part")
sulfuric_acid= item(53, {itemTag.chemical},
                        {itemFlag.isReactive, itemFlag.isLiquid, itemFlag.isToxic},                           4, False, "sulfuric_acid")
nitroglycerin_item = item(54, {itemTag.nitroglycerin, itemTag.chemical, itemTag.explosive},
                        {itemFlag.isReactive, itemFlag.isLiquid,
                         itemFlag.isFlammable, itemFlag.isToxic, itemFlag.isValuable},                        4, False, "nitroglycerin")
rubber_item  = item(55, {itemTag.rubber},
                        {itemFlag.isFlexible, itemFlag.isDurable},                                            4, True,  "rubber")
aniline_dye  = item(56, {itemTag.dye, itemTag.chemical},
                        {itemFlag.isLiquid, itemFlag.isValuable, itemFlag.isReactive},                        4, False, "aniline_dye")

# ── Late Industrial (level 5) ─────────────────────────────────────────────────
dynamite     = item(57, {itemTag.explosive, itemTag.chemical},
                        {itemFlag.isReactive, itemFlag.isValuable, itemFlag.isDurable},                       5, False, "dynamite")
copper_wire  = item(58, {itemTag.metal},
                        {itemFlag.isConductive, itemFlag.isFlexible, itemFlag.isValuable},                    5, False, "copper_wire")
steam_engine = item(59, {itemTag.machinery},
                        {itemFlag.isDurable, itemFlag.isValuable, itemFlag.isBulky},                          5, False, "steam_engine")
telegraph    = item(60, {itemTag.machinery},
                        {itemFlag.isDurable, itemFlag.isConductive,
                         itemFlag.isValuable, itemFlag.isLuxury},                                             5, False, "telegraph")
refined_oil  = item(61, {itemTag.fuel},
                        {itemFlag.isFlammable, itemFlag.isLiquid,
                         itemFlag.isHighCalorie, itemFlag.isValuable},                                        5, False, "refined_oil")

# ── Register all authored items ───────────────────────────────────────────────
_all_items = [
    wheat, rye, bread, wool_raw, woolen_fabric, basic_clothes, timber, charcoal,
    iron_ore, iron_ingot, iron_tool, iron_sword, hide, leather_item, salt,
    clay_pots, ale, mead, wax_candle, cotton_raw, cotton_fabric, fine_clothes,
    flax, linen_fabric, copper_ore, copper_ingot, tin_ore, bronze_ingot,
    crossbow, arrows, indigo_dye, parchment, quarry_stone, brick, tar,
    silk_thread, silk_fabric, silk_clothes, steel_ingot, steel_tool, steel_sword,
    saltpeter_item, sulfur_item, gunpowder, musket_ball, musket, glass_item, rag_paper,
    coal_item, coke_item, cast_iron, machine_part, sulfuric_acid, nitroglycerin_item,
    rubber_item, aniline_dye, dynamite, copper_wire, steam_engine, telegraph, refined_oil,
]
for _i in _all_items:
    globalItemIndex[_i.id] = _i


# =============================================================================
# PRODUCTION RULES
# =============================================================================

# Crops → food
registerProductionRule(productionRule(
    [itemTag.crops], itemTag.food, 1,
    inputPreferences={itemTag.crops: [itemFlag.isNutritional]},
    qualityTiers=[
        qualityTier(1.0, {itemFlag.isNutritional, itemFlag.isValuable}),
        qualityTier(0.0, {itemFlag.isNutritional}),
    ]
))

# Fiber → fabric
registerProductionRule(productionRule(
    [itemTag.fiber], itemTag.fabric, 1,
    inputPreferences={itemTag.fiber: [itemFlag.isSmooth, itemFlag.isFlexible]},
    qualityTiers=[
        qualityTier(1.0, {itemFlag.isSmooth, itemFlag.isFlexible, itemFlag.isLuxury}),
        qualityTier(0.5, {itemFlag.isSmooth, itemFlag.isFlexible}),
        qualityTier(0.0, {itemFlag.isFlexible}),
    ]
))

# Fabric → consumable (clothing)
registerProductionRule(productionRule(
    [itemTag.fabric], itemTag.consumable, 1,
    inputPreferences={itemTag.fabric: [itemFlag.isSmooth, itemFlag.isFlexible, itemFlag.isLuxury]},
    qualityTiers=[
        qualityTier(0.8, {itemFlag.isSmooth, itemFlag.isValuable, itemFlag.isLuxury}),
        qualityTier(0.4, {itemFlag.isDurable, itemFlag.isValuable}),
        qualityTier(0.0, {itemFlag.isDurable}),
    ]
))

# Ore + fuel → metal (basic smelting)
registerProductionRule(productionRule(
    [itemTag.ore, itemTag.fuel], itemTag.metal, 2,
    inputPreferences={
        itemTag.fuel: [itemFlag.isHighCalorie, itemFlag.isFlammable],
        itemTag.ore:  [itemFlag.isPure],
    },
    qualityTiers=[
        qualityTier(0.75, {itemFlag.isRefined, itemFlag.isPure}),
        qualityTier(0.0,  {itemFlag.isRefined}),
    ]
))

# Ore + coke → metal (blast furnace)
registerProductionRule(productionRule(
    [itemTag.ore, itemTag.coke], itemTag.metal, 3,
    inputPreferences={
        itemTag.coke: [itemFlag.isHighCalorie],
        itemTag.ore:  [itemFlag.isPure],
    },
    qualityTiers=[
        qualityTier(0.75, {itemFlag.isRefined, itemFlag.isPure, itemFlag.isHardened}),
        qualityTier(0.0,  {itemFlag.isRefined, itemFlag.isHardened}),
    ]
))

# Metal + fuel → tools
registerProductionRule(productionRule(
    [itemTag.metal, itemTag.fuel], itemTag.tools, 2,
    inputPreferences={itemTag.metal: [itemFlag.isHardened, itemFlag.isPure, itemFlag.isAlloyed]},
    qualityTiers=[
        qualityTier(0.8, {itemFlag.isDurable, itemFlag.isHardened, itemFlag.isValuable}),
        qualityTier(0.4, {itemFlag.isDurable, itemFlag.isValuable}),
        qualityTier(0.0, {itemFlag.isDurable}),
    ]
))

# Metal + fuel → weapons
registerProductionRule(productionRule(
    [itemTag.metal, itemTag.fuel], itemTag.weapons, 2,
    inputPreferences={itemTag.metal: [itemFlag.isHardened, itemFlag.isPure, itemFlag.isAlloyed]},
    qualityTiers=[
        qualityTier(0.8, {itemFlag.isDurable, itemFlag.isHardened, itemFlag.isValuable}),
        qualityTier(0.0, {itemFlag.isDurable}),
    ]
))

# Saltpeter + sulfur + crops → explosive (gunpowder)
registerProductionRule(productionRule(
    [itemTag.saltpeter, itemTag.sulfur, itemTag.crops], itemTag.explosive, 3,
    inputPreferences={
        itemTag.saltpeter: [itemFlag.isPowdered, itemFlag.isPure],
        itemTag.sulfur:    [itemFlag.isPowdered, itemFlag.isPure],
    },
    qualityTiers=[
        qualityTier(0.8, {itemFlag.isReactive, itemFlag.isValuable, itemFlag.isPowdered}),
        qualityTier(0.0, {itemFlag.isReactive, itemFlag.isPowdered}),
    ]
))

# Nitroglycerin → explosive (dynamite)
registerProductionRule(productionRule(
    [itemTag.nitroglycerin, itemTag.miscellaneous], itemTag.explosive, 5,
    inputPreferences={itemTag.nitroglycerin: [itemFlag.isPure]},
    qualityTiers=[
        qualityTier(1.0, {itemFlag.isReactive, itemFlag.isValuable, itemFlag.isDurable}),
        qualityTier(0.0, {itemFlag.isReactive, itemFlag.isValuable}),
    ]
))

# Coal + fuel → coke
registerProductionRule(productionRule(
    [itemTag.coal, itemTag.fuel], itemTag.coke, 4,
    inputPreferences={itemTag.fuel: [itemFlag.isHighCalorie]},
    qualityTiers=[
        qualityTier(1.0, {itemFlag.isRefined, itemFlag.isHighCalorie}),
        qualityTier(0.0, {itemFlag.isRefined}),
    ]
))

# Metal + metal → alloy (steelmaking)
registerProductionRule(productionRule(
    [itemTag.metal, itemTag.metal], itemTag.alloy, 3,
    inputPreferences={itemTag.metal: [itemFlag.isPure, itemFlag.isRefined]},
    qualityTiers=[
        qualityTier(1.0, {itemFlag.isAlloyed, itemFlag.isHardened, itemFlag.isPure, itemFlag.isValuable}),
        qualityTier(0.5, {itemFlag.isAlloyed, itemFlag.isHardened, itemFlag.isValuable}),
        qualityTier(0.0, {itemFlag.isAlloyed, itemFlag.isValuable}),
    ]
))

# Metal + machinery → machinery
registerProductionRule(productionRule(
    [itemTag.metal, itemTag.machinery], itemTag.machinery, 4,
    inputPreferences={itemTag.metal: [itemFlag.isHardened, itemFlag.isPure]},
    qualityTiers=[
        qualityTier(0.8, {itemFlag.isDurable, itemFlag.isValuable}),
        qualityTier(0.0, {itemFlag.isDurable}),
    ]
))

# ── Tannery ───────────────────────────────────────────────────────────────────
registerProductionRule(productionRule(
    [itemTag.livestock, itemTag.fuel], itemTag.leather, 1,
    inputPreferences={itemTag.livestock: [itemFlag.isRough, itemFlag.isDurable]},
    qualityTiers=[
        qualityTier(0.7, {itemFlag.isDurable, itemFlag.isFlexible, itemFlag.isRefined}),
        qualityTier(0.0, {itemFlag.isDurable, itemFlag.isFlexible}),
    ]
))
registerProductionRule(productionRule(
    [itemTag.leather], itemTag.consumable, 1,
    inputPreferences={itemTag.leather: [itemFlag.isDurable, itemFlag.isFlexible]},
    qualityTiers=[
        qualityTier(0.6, {itemFlag.isDurable, itemFlag.isValuable}),
        qualityTier(0.0, {itemFlag.isDurable}),
    ]
))
registerProductionRule(productionRule(
    [itemTag.leather, itemTag.metal], itemTag.armor, 2,
    inputPreferences={
        itemTag.leather: [itemFlag.isDurable, itemFlag.isTough],
        itemTag.metal:   [itemFlag.isHardened, itemFlag.isDurable],
    },
    qualityTiers=[
        qualityTier(0.8, {itemFlag.isDurable, itemFlag.isHardened, itemFlag.isValuable}),
        qualityTier(0.3, {itemFlag.isDurable, itemFlag.isValuable}),
        qualityTier(0.0, {itemFlag.isDurable}),
    ]
))

# ── Brewery ───────────────────────────────────────────────────────────────────
registerProductionRule(productionRule(
    [itemTag.crops, itemTag.fuel], itemTag.alcohol, 1,
    inputPreferences={
        itemTag.crops: [itemFlag.isNutritional],
        itemTag.fuel:  [itemFlag.isFlammable],
    },
    qualityTiers=[
        qualityTier(0.5, {itemFlag.isNutritional, itemFlag.isValuable}),
        qualityTier(0.0, {itemFlag.isNutritional}),
    ]
))
registerProductionRule(productionRule(
    [itemTag.crops, itemTag.spice], itemTag.alcohol, 2,
    inputPreferences={
        itemTag.crops: [itemFlag.isNutritional],
        itemTag.spice: [itemFlag.isValuable, itemFlag.isRare],
    },
    qualityTiers=[
        qualityTier(0.9, {itemFlag.isNutritional, itemFlag.isValuable, itemFlag.isLuxury}),
        qualityTier(0.4, {itemFlag.isNutritional, itemFlag.isValuable}),
        qualityTier(0.0, {itemFlag.isNutritional}),
    ]
))

# ── Kiln ──────────────────────────────────────────────────────────────────────
registerProductionRule(productionRule(
    [itemTag.stone, itemTag.fuel], itemTag.ceramic, 1,
    inputPreferences={itemTag.fuel: [itemFlag.isHighCalorie, itemFlag.isFlammable]},
    qualityTiers=[
        qualityTier(0.6, {itemFlag.isDurable, itemFlag.isTough}),
        qualityTier(0.0, {itemFlag.isDurable}),
    ]
))
registerProductionRule(productionRule(
    [itemTag.stone, itemTag.lumber], itemTag.stone, 2,
    inputPreferences={itemTag.lumber: [itemFlag.isDurable]},
    qualityTiers=[
        qualityTier(0.7, {itemFlag.isDurable, itemFlag.isTough, itemFlag.isRefined}),
        qualityTier(0.0, {itemFlag.isDurable, itemFlag.isTough}),
    ]
))
registerProductionRule(productionRule(
    [itemTag.ceramic, itemTag.fuel], itemTag.glass, 3,
    inputPreferences={itemTag.fuel: [itemFlag.isHighCalorie]},
    qualityTiers=[
        qualityTier(0.8, {itemFlag.isFragile, itemFlag.isValuable, itemFlag.isRefined}),
        qualityTier(0.0, {itemFlag.isFragile, itemFlag.isValuable}),
    ]
))

# ── Lumber / Paper ────────────────────────────────────────────────────────────
registerProductionRule(productionRule(
    [itemTag.lumber, itemTag.fuel], itemTag.fuel, 1,
    inputPreferences={itemTag.lumber: [itemFlag.isDurable]},
    qualityTiers=[
        qualityTier(0.5, {itemFlag.isFlammable, itemFlag.isHighCalorie}),
        qualityTier(0.0, {itemFlag.isFlammable}),
    ]
))
registerProductionRule(productionRule(
    [itemTag.fiber, itemTag.lumber], itemTag.paper, 2,
    inputPreferences={itemTag.fiber: [itemFlag.isSmooth, itemFlag.isFlexible]},
    qualityTiers=[
        qualityTier(0.7, {itemFlag.isDurable, itemFlag.isFragile}),
        qualityTier(0.0, {itemFlag.isFragile}),
    ]
))
