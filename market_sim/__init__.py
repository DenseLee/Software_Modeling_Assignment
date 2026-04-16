"""market_sim — backward-compatible public API.

Importing this package re-exports every symbol that external code
(train.py, ServerTest.py, sim_analysis.py) expects to find under
the ``market_sim`` namespace.
"""
from .types       import (ActionType, ActionParamSpec, ACTION_SPECS,
                           itemTag, itemFlag, item, qualityTier, ActionResult, TradeDeal)
from .structures  import (productionRule, productionGraphs, registerProductionRule,
                           ProductionChain, BuildingType, Building, Market, Infrastructure,
                           itemTrackUnit, ALL_BUILDING_TYPES,
                           BT_FARM, BT_FORGE, BT_WORKSHOP, BT_MILL,
                           BT_TANNERY, BT_BREWERY, BT_KILN, BT_ALCHEMIST, BT_FACTORY,
                           MAX_TECH_LEVEL, TECH_TIER_NAMES, PopChangeEvent, PopClass)
from ._globals    import globalItemIndex, globalControllers, registerController
from .controller  import MarketController
from .world       import *   # triggers item + rule registration at import time
from .production  import (INHERITABLE_FLAGS, FLAG_CONFLICTS, FLAG_MUTATIONS,
                           inheritFlags, simulateProduction, requestProductionChainCreation,
                           scoreItemForSlot, SpawnProfile, SPAWN_PROFILES, trySpawnItems)
from .seasons     import (getCurrentSeason, getEffectiveSeason, SEASON_NAMES,
                           TICKS_PER_SEASON, TICKS_PER_YEAR, FARM_SEASON_MULT,
                           SEASON_FOOD_CONSUMPTION_MULT, BUILDING_WINTER_MULT)
from .tick        import (computeCoinValue, updateMarketPrices, globalTick,
                           tickInfrastructureDrain, tickLogisticsInput, tickLogisticsOutput,
                           tickSubsistence, tickConsumption, tickPopQueue,
                           tickProductionProgress, tickWageBalancing, tickBuildingViability,
                           tickProduction, tickPopulation, randomEventFunction)
from .metrics     import (computeGDP, computeDiversityScore, computeUnemploymentRate,
                           computeAvgHappiness, computeAvgLiteracy, computeTierRatio,
                           computeProductionEfficiency, computeEventPenalty,
                           computeEventMitigationScore, computeReward,
                           computePrivateProfit, computeEconomicStrength,
                           computeSupplyChainBalance, computeTaxBurden)
from .taxation    import (TaxPolicy, TaxRecord, tickTaxCollection,
                           TAX_POLICY_EARLY, TAX_POLICY_MID, TAX_POLICY_LATE)
from .setup       import bootstrapEconomy
from .environment import (STATE_DIM, ACTION_SHAPE, N_ACTIONS, MAX_BUILDINGS,
                           MarketEnvironment, SingleAgentEnv, makeSingleAgentEnvs,
                           computeActionMask)
from .ai_agents   import LocalAI, RulerAI
from .heuristic   import runTestSimulation
